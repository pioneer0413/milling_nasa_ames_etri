# ICTC 2026 — Research Context

ICTC 2026 논문 작업의 Notion 마스터 페이지와 하위 페이지 맥락을 repo에 고정한 문서.
어느 디바이스에서 clone해도 연구 배경/가설/도메인 지식을 Notion 접근 없이 파악할 수 있게 한다.
(스냅샷 기준일: 2026-06-22. Notion이 최신 원본이며, 본 문서는 요약·맥락 보존용.)

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
- **Run-sequence Modeling** DB: 활성 실험 (B/I/E class). B(Baseline) 4건, I(Investigation) 5건, E(Extension/Evaluation) 7건. 옛 H1~H9 가설 번호는 Archive로 이동했으며, 일부는 이 체계로 재편됨(예: 구 H7 Case1/Case12 분석 → E3/E4).
- **Analysis** DB: 분석 노트, Run-sequence Modeling과 관계형 연결. 예) "PHM2010에서 run-sequence가 run-independent에 뒤지는 이유".
- **Model** DB: 모델 정의/설정 카탈로그, Run-sequence Modeling과 관계형 연결.

### 3단계 confound-free 검증 구조 (2026-06-22 사용자 재구성)

Segment-aware 입력 선택과 Run-sequence Modeling을 독립적인 두 기여 요인으로 분리 입증하기 위해 **B1/B2/B3을 새로 정의**(Outline → Method Overview에도 동일 구조 기록):

| 단계 | 항목 | Goal (현재 Notion 원문) | 상태 |
|---|---|---|---|
| 1) Segment × Model | **[B1] Model x (Entry-Steady)** | Entry-Steady 입력(Feature, Raw signal)에 대해 모든 모델(CF/Ridge/RF/XGBoost/SVR/Feature-MLP/Feature-GRU·LSTM/Signal-CNN/Signal-GRU·LSTM) 간 성능 비교 | 시작 전 (계획 작성됨; 10개 모델 **전부** 신규 실행 필요 — Carry-forward 제외, 아래 Steady 오염 항목 참고) |
| 2) Segment-ablation | **[B2] Ablation: Segment** | 모델을 FeatGRU/LSTM·XGBoost·RF로 고정하고 Segment(Full/Exclude No-load/Entry/Entry-steady/Steady)별 성능 비교 | 시작 전 (내용 비어있음) |
| 3) Model-comparison | **[B3] Comparison: Model** | Segment=Full 고정, Feature-based GRU/LSTM·Signal-based GRU/LSTM 간 성능 비교 | 시작 전 (내용 비어있음, 비-시퀀스 baseline 미포함) |

**Segment 정의 단일화 (2026-06-22 결정, 해결됨)**: segment 인덱스 파일 난립 문제 해소 — 이후 모든 B/I/E 실험은 **`datasets/nasa/cutting_segment_index.csv` 단일 파일**만 사용하고 Steady/Steady_v2 구분을 폐지한다. 컬럼 기반 정의: `Full=[0, signal_length)`, `Excl_Exit=[0, idx_exit_start)`, `Entry_Steady=[idx_noload_end, idx_exit_start)`, `Steady=[idx_start, idx_exit_start)`(`idx_exit_start` = peng2026_inspired exit 검출 경계만 사용). [B1]/[B2]의 `Exclude No-load`/`Entry-steady` 등 명칭은 이 4종 중 하나로 매핑해서 통일.

**Steady 경계 오염 — Entry_Steady 결과 전면 폐기 (2026-06-23)**: cutting_segment_index.csv 단일화 과정에서 기존 `idx_start`(Steady 시작 경계)가 오염된 것으로 확인됨. [I3] Stage A/C의 Entry_Steady/Steady/Steady_v2 RMSE(0.083903, 0.086679, 0.118339, 0.144295, 0.119317, 0.141702 등)는 전부 폐기 — 인용·재사용 금지. [B1]의 "재사용 가능한 기존 결과"에서 Feature-GRU/LSTM 항목 제거, 신규 실행 목록으로 이동(Carry-forward만 영향 없음). [I4]에도 동일 경고 반영됨. 경계 정리 후 [B1]/[I3]/[I4] 전체 재실행 필요.

**남은 미해결 항목**:
- [B3] Goal에 RF/XGBoost/SVR/MLP/SignalCNN 등 비-시퀀스 baseline이 빠져 있음 — "Model-comparison" 역할을 채우려면 추가 필요.
- 기존 [I3]/[I4]는 그대로 유지된 채 새 B1/B2/B3과 역할이 겹침(I4의 Entry_Steady 열 ≈ B1, I3 Stage A ≈ B2 일부, 구 B3 종합 baseline ≈ B3) — 통합 또는 역할 분담 정리 필요.

### ⚠️ 데이터 보존 경고: 구 [B3]/[B4] 페이지 삭제(trash)됨

번호 재배치 과정에서 기존 두 페이지가 Notion에서 **삭제(trash) 처리**됨(`<page ... deleted>` 상태로만 fetch 가능, 휴지통 보존기간 내 복구 가능):
- 구 **[B3] Naive Baseline — Carry-Forward & Linear Models**: 13개 모델 종합 비교(FeatGRU 0.0951, FeatLSTM 0.0922, RF 0.1292, XGBoost ref 0.1092, Cai2020 0.0626 등) + 한계/TODO 전체.
- 구 **[B4] 5-Seed Stability**: H17_S1 + H22_S1 확장 결과(FeatLSTM/RF/MLP_Feat/SignalCNN 5-seed CV 표).
- 새 **[B4]**는 구 [B2] Shuffled-Sequence Ablation 자리를 재사용한 것이며, 상태가 "완료"→**"시작 전"으로 리셋**됨(2026-06-22, 사용자 확인).

위 두 페이지의 수치는 본 문서 §4·§3(배제 접근)에 이미 반영되어 있어 정보 자체는 보존되지만, **Notion 원본 페이지로서는 사라진 상태** — 새 [B3] Comparison: Model을 채울 때 이 데이터를 가져와 재사용하는 것을 권장.

### Performance Objective (현재 best)
| 데이터셋 | 목표 (LOCV RMSE) | 달성 | Best |
|---|---|---|---|
| NASA Ames | ≤ 0.080 mm | No | FeatLSTM 0.092217 |
| PHM2010 | ≤ 10 µm | No | FeatLSTM 22.04 (raw feature 기준) |

> 단, E5_S1(Delta feature)에서 PHM2010 sequence 모델이 RF를 역전 (FeatLSTM delta 20.23 / FeatGRU raw+delta 20.39µm) — 본 repo 실험 라인 참조.

### NASA 앙상블 라인 (B5 / B6)

단일 모델 한계를 넘기 위한 앙상블 실험. 모두 LOCV-15 observed-VB-only RMSE.

| 실험 | 방법 | RMSE | 비고 |
|---|---|---|---|
| B5_S2 | 4-ML 단순 평균 (own-best subset) | 0.103189 | ML만 |
| B5_S3 | FeatGRU 메타러너 스태킹 | — | nested-LOCV |
| **B6_S1** | **ID-SWE (E0 평균 앙상블)** | **0.096644** | GRU+LSTM+XGB+RF+SVR, 5종 평균 |
| B6_S1 | ID-SWE E2/E3 (disagreement 적응 가중) | 0.096961 / 0.097612 | **E0와 동률 — 게이팅 무이득** |

> **B6_S1 결론 (네거티브/중립)**: disagreement 기반 적응 게이팅(E2/E3)이 단순 평균(E0) 대비 개선 없음. 원인 = 이 구성의 sequence base(GRU 0.116/LSTM 0.118)가 ML(XGB/SVR 0.109)보다 약한데 게이트 가중 `w∈[0.40,0.75]`이 sequence를 항상 우대해 약한 그룹을 과신. 단, E3(median-other)는 late-phase·Case 1/12/13에서 소폭 우위로 "후반 과상승 억제" 방향성은 일부 확인. 5종 평균 앙상블 자체는 모든 단일 base 및 B5_S2(0.1032)를 능가하나 튜닝 단일 best(FeatLSTM 0.0922)에는 못 미침. 스크립트: `scripts/run_B6_S1_inverse_disagreement_sequence_weighted_ensemble.py`.

### 비교 대상에서 배제된 접근 (2026-06-22 결정)

수치상 더 낮은 RMSE를 보였더라도 아래 두 가지는 메인 leaderboard/논문 비교에서 배제한다:

- **Prefix 비율 고정값 (H12~H16, 예: GRU prefix=80% → 0.081977)**: sweep 결과 인접 비율(70%→0.098, 90%→0.093)과 단절된 단일 지점이며, 고정 상수가 다른 공정조건·신호 길이에 일반화된다는 보장이 없어 실제 배포 환경에 부적합. Segment-aware(entry/steady/exit 식별) 대안도 현재 구현(I3/I4)에서는 동등 비교가 끝나지 않은 상태.
- **Cai2020 hybrid LSTM (RMSE 0.0626, 1-seed)**: 이전 run의 **실측(ground-truth) VB**를 test 시에도 입력 feature로 사용하는 구조 — 실제 배포 시점에는 알 수 없는 값을 안다고 가정하는 비현실적 설정이므로 배제. 이 모델 제외 후 NASA 공식 best는 FeatLSTM 0.092217로 유지된다. (원 기록처였던 Notion 구-[B3] 페이지는 2026-06-22 번호 재배치 중 삭제됨 — §3 "데이터 보존 경고" 참고)

---

## 4. 핵심 인사이트 (Idea)

- **선형 보간 VB**는 모델 공통적으로 성능 향상에 도움 ⭐.
- `run=1`의 VB=0은 유효하지 않음 → 전처리에서 제거 후 학습.
- **목표 RMSE는 0.1 mm 이내**여야 실질적으로 유효한 예측 ⭐ (RMSE 스케일 해석 표: ≤0.03 매우 우수 … ≥0.10 주장 조심).
- **Critical Mistake 1** ⭐: H1_S1/S2/S3 집계 시 **sensor에 대해 평균**을 내버려 센서별 feature/segment 특성 해석이 왜곡됨(`feature_segment_rank_average`류 표가 smcAC/smcDC의 강점을 희석). sensor-level 원본 산출물에는 정보 보존.
- **어려운 Case**: Case 13/14(신호 패턴 상이 → 난이도↑), Case 1/12 근본 원인은 H7_S1에서 규명:
  - **Case 1**: 유일한 non-monotone VB 궤적(Run15 peak→Run17 감소) → GRU 단조증가만 학습 → 파국적 오차. GRU/XGB 3.1배. (피처 문제 아님, OOD 아님)
  - **Case 12**: Delta baseline 분포 편이(`(12,1)` 제외로 기준=Run2 VB=0.05) → 두 모델 동등 실패 = 피처 파이프라인 결함.
- **누적 절삭 시간(time) meta feature는 기여 없음** (H8_S1~S3, 2026-06-12, `scripts/run_H8_S{1,2,3}_*.py`):
  - `time_norm`(case-wise 정규화) 추가 시 GRU −6.89%/XGB −11.43% 개선되어 보였으나, leakage 효과로 확인됨(case 마지막 런 시간을 알아야 계산 가능).
  - leakage-free 버전인 `raw_time`/`elapsed_time`은 GRU +54.8%/+60.0% 악화, XGB는 무변화 — 시간 정보는 실질 신호 없음.
  - **결론**: `{DOC, feed, material}` 3개 meta feature로 충분, time 계열 추가 불필요.
- **표현학습 PoC 진행 중 ([I5], 보완 필요)**: VB-feature locality 비대칭(VB→feature corr 0.595, 약함) 보강을 위해 FiLM conditioning + Rank-N-Contrast(RNC) 구현·smoke test 통과. I3 best(FeatGRU 0.0839 @ Entry_Steady)의 0.080 돌파 목표 — **단, 이 0.0839 수치는 2026-06-23 폐기됨(아래 참고), 목표 기준 재확정 필요**. LOCV 실측은 아직 미수행. **FiLM 부분은 PoC를 넘어 정식 모델로 코드화됨** — `feature_film_gru`(메타 전용 FiLM, signal feature는 불변)가 `feature_gru_regressor.py`/`h2_regressors.py`에 등록되고 유닛테스트 통과(`run_H4_S5_feature_film_gru_segments.py`, `run_H4_S5_feature_gru_hpo.py --use_metadata_film`).
- **⚠️ Entry_Steady 결과 전부 폐기 (2026-06-23)**: [I3] Stage A/C에서 사용한 Steady 경계(`idx_start`)가 오염된 것으로 확인됨. Entry_Steady/Steady/Steady_v2 관련 모든 RMSE(0.083903, 0.086679, 0.118339, 0.144295, 0.119317, 0.141702 등)는 인용·재사용 금지. `datasets/nasa/cutting_segment_index.csv` 기준으로 [B1]/[I3]/[I4] 전체 재실행 필요. [B1]의 "재사용 가능한 기존 결과"에서 Feature-GRU/LSTM 항목은 제거되고 신규 실행 목록으로 이동함(Carry-forward만 segment 비의존이라 영향 없음).
- **실시간성(향후)**: 전체 cut 신호 입력 → cut 완료 후 예측. 실제 공정은 다음 cut 전 교체 결정 필요 → cut 간격 내 추론 성능도 만족해야 함.
- **Segment 정의 단일화 (2026-06-22 결정)**: segment 관련 인덱스 파일이 V1/V2 등으로 난립해 혼선이 있었음 → **`datasets/nasa/cutting_segment_index.csv` 단일 파일만 사용**, Steady/Steady_v2 구분 폐지. 컬럼 기반 정의: `Full=[0, signal_length)`, `Excl_Exit=[0, idx_exit_start)`, `Entry_Steady=[idx_noload_end, idx_exit_start)`, `Steady=[idx_start, idx_exit_start)` — `idx_exit_start`(peng2026_inspired exit 검출)만 사용, 옛 `idx_end`/V1 경계는 더 이상 segment 정의에 쓰지 않음.
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
- `scripts/run_B1_S1_phm2010_feature_baseline.py` — 9-model LOCV-3 baseline. **주의**: 파일명의 "B1"은 옛 번호(스크립트 작성 시점). 2026-06-22 Notion 재배치로 현재는 **[E7] PHM2010 Feature-GRU LOCV 실행**에 해당 — 새 Notion [B1]은 NASA "Model x (Entry-Steady)"로 의미가 다름.
- `scripts/run_B1_S2_phm2010_flute_augment.py` — flute-label 증강(궤적 2→6). 위와 동일하게 파일명 번호는 옛 [B1](현 [E7]) 기준.
- `scripts/run_E5_S1_phm2010_delta_feature.py` — [E5] Delta feature 비대칭 해소(sequence가 RF 역전).

NASA 라인(H 시리즈)은 `datasets/nasa/raw_signal.csv`(gitignored)가 필요. 자세한 데이터 가용성은 `README.md` 참조.
